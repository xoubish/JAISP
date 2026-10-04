import unittest
import numpy as np
import torch
from scipy.special import erf
from models.photometry.self_supervised.core import BANDS, MorphologyHead, templates, fit_flux
from models.photometry.self_supervised.data import jacobian
from astropy.wcs import WCS


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_bands_independent_blended_fluxes(self):
        # Exact separable Gaussian pixel integrals are independent of our renderer.
        # Both resolutions, fractional positions, strongly overlapping neighbors,
        # signed faint amplitudes, free sky, and different colors in every band.
        for i,band in enumerate(BANDS):
            scale=.2 if i<6 else .1
            psf=2. if i<6 else (1. if i==6 else 2.)
            positions=torch.tensor([[19.23,20.47],[21.61,20.16]],dtype=torch.float64)
            intrinsic=np.array([.12,.22])
            cov=torch.tensor(np.array([np.eye(2)*s*s for s in intrinsic]))
            flux=torch.tensor([150.+i*13,80.-i*12],dtype=torch.float64)
            sigma=np.sqrt((intrinsic/scale)**2+psf**2)
            grid=np.arange(43)
            exact=[]
            for xy,s in zip(positions.numpy(),sigma):
                px=.5*(erf((grid+.5-xy[0])/(np.sqrt(2)*s))-erf((grid-.5-xy[0])/(np.sqrt(2)*s)))
                py=.5*(erf((grid+.5-xy[1])/(np.sqrt(2)*s))-erf((grid-.5-xy[1])/(np.sqrt(2)*s)))
                exact.append(np.outer(py,px).ravel())
            image=(torch.tensor(np.stack(exact,1))@flux+2.7).reshape(43,43)
            t=templates(positions,cov,torch.eye(2,dtype=torch.float64)/scale,psf,image.shape,oversample=3)
            r=fit_flux(image,torch.ones_like(image),t)
            np.testing.assert_allclose(r['flux'],flux,atol=.005,rtol=1e-4,err_msg=band)
            self.assertAlmostEqual(float(r['background']),2.7,places=3)

    def test_flux_normalization_and_crop_wings(self):
        pos=torch.tensor([[20.1,20.7]],dtype=torch.float64)
        cov=torch.eye(2,dtype=torch.float64)[None]*4
        large=templates(pos,cov,torch.eye(2,dtype=torch.float64),1.,(50,50))
        small=templates(pos,cov,torch.eye(2,dtype=torch.float64),1.,(21,21))
        self.assertAlmostEqual(float(large.sum()),1.,places=6)
        self.assertLess(float(small.sum()),.4)

    def test_masked_invalid_pixels_and_conditional_covariance(self):
        t=torch.tensor([[1.,0.],[0.,1.],[.2,.3],[.1,.2],[.3,.1],[.4,.2]],dtype=torch.float64)
        y=t@torch.tensor([7.,-2.],dtype=torch.float64)+3
        v=torch.ones_like(y);v[0]=float('nan');y[1]=float('nan')
        r=fit_flux(y,v,t)
        np.testing.assert_allclose(r['flux'],[7.,-2.],atol=1e-9)
        a=torch.cat((t[2:],torch.ones(4,1,dtype=torch.float64)),1)
        np.testing.assert_allclose(r['covariance'],np.linalg.inv(a.T@a)[:2,:2],rtol=1e-10)

    def test_singular_blend_flagged(self):
        t=torch.ones(10,2)
        with self.assertRaisesRegex(ValueError,'Degenerate'):
            fit_flux(torch.ones(10),torch.ones(10),t)

    def test_gradient_through_joint_fit_and_frozen_features(self):
        torch.manual_seed(3)
        head=MorphologyHead(channels=2).double()
        feature=torch.randn(2,2,3,3,dtype=torch.float64,requires_grad=True)
        base=torch.eye(2,dtype=torch.float64).repeat(2,1,1)*.04
        positions=torch.tensor([[10.2,10.4],[12.7,10.8]],dtype=torch.float64)
        j=torch.eye(2,dtype=torch.float64)*5
        truth=templates(positions,base*1.5,j,1.5,(25,25))@torch.tensor([100.,70.],dtype=torch.float64)
        cov=head(feature,base)
        t=templates(positions,cov,j,1.5,(25,25))
        loss=fit_flux(truth,torch.ones_like(truth),t)['loss'];loss.backward()
        self.assertIsNone(feature.grad)
        self.assertGreater(float(head.net[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
        analytic=float(head.net[-1].bias.grad[0])
        def value(delta):
            with torch.no_grad():head.net[-1].bias[0]=delta
            t=templates(positions,head(feature,base),j,1.5,(25,25))
            return float(fit_flux(truth,torch.ones_like(truth),t)['loss'])
        numerical=(value(1e-5)-value(-1e-5))/2e-5
        self.assertAlmostEqual(analytic,numerical,places=7)

    def test_noise_coverage_in_crowded_scene(self):
        # A Monte Carlo check of deblending covariance, not just noiseless inversion.
        torch.manual_seed(37)
        dtype=torch.float64
        positions=torch.tensor([[9.3,10.1],[11.0,10.7]],dtype=dtype)
        cov=torch.eye(2,dtype=dtype).repeat(2,1,1)*.04
        t=templates(positions,cov,torch.eye(2,dtype=dtype)*5,2.,(23,23))
        flux=torch.tensor([80.,15.],dtype=dtype)
        signal=t@flux+4.
        pulls=[]
        for _ in range(300):
            noise=torch.randn(signal.shape,dtype=dtype)*.3
            result=fit_flux(signal+noise,torch.full_like(signal,.09),t)
            pulls.append(((result['flux']-flux)/result['error']).numpy())
        pulls=np.array(pulls)
        self.assertTrue(np.all(np.abs(pulls.mean(0))<.15))
        self.assertTrue(np.all(np.abs(pulls.std(0)-1)<.15))
        covered=(np.abs(pulls)<1).mean(0)
        self.assertTrue(np.all((covered>.60)&(covered<.76)))
        self.assertLess(float(result['covariance'][0,1]),0)

    def test_rotated_wcs_jacobian(self):
        wc=WCS(naxis=2);wc.wcs.ctype=['RA---TAN','DEC--TAN'];wc.wcs.crval=[53.,-28.]
        wc.wcs.crpix=[32.,32.]
        theta=.7
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        wc.wcs.cd=rotation*.2/3600
        actual=jacobian(wc,np.array([31.,31.]))
        np.testing.assert_allclose(actual,rotation*.2,atol=1e-7)

if __name__=='__main__':unittest.main()
