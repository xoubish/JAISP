import unittest
import numpy as np
import torch
from models.photometry.self_supervised.core import BANDS
from models.photometry.self_supervised.mixture import (
    SCALES,dictionary,positive_profile_fit,signed_measurement,fit_multiband)
from models.photometry.self_supervised.injections import exponential_profile


class MixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def scene(self, count=1):
        bands={}
        for b in BANDS:
            scale=.2 if b.startswith('rubin') else .1
            pos=torch.tensor([[30.,30.],[34.,31.]])[:count]
            bands[b]=dict(image=torch.zeros(61,61),variance=torch.ones(61,61)*.01,
                          mask=torch.ones(61,61,dtype=torch.bool),positions=pos,
                          sky_to_pixel=torch.eye(2)/scale,psf_sigma=1.5)
        return dict(bands=bands,sky=np.zeros((count,2)),central=0)

    def test_profile_regularization_does_not_set_amplitude(self):
        scene=self.scene(2);d=scene['bands']['euclid_VIS']
        bank=dictionary(scene,'euclid_VIS',torch.eye(2).repeat(2,1,1))
        mixture=np.zeros((2,len(SCALES)));mixture[:,[1,3]]=[[.2,.8],[.7,.3]]
        truth=np.array([317.,91.])
        image=np.einsum('pnk,nk,n->p',bank,mixture,truth).reshape(61,61)-8.
        d['image']=torch.tensor(image,dtype=torch.float64)
        # Reference amplitudes are deliberately wrong by orders of magnitude.
        weights,total=positive_profile_fit(d,bank,mixture,strength=1e5,reference_flux=np.array([1.,10000.]),precision=np.diag(np.geomspace(.1,100,len(SCALES))))
        np.testing.assert_allclose(total,truth,rtol=1e-5,atol=.001)
        measured=signed_measurement(d,bank,weights)
        np.testing.assert_allclose(measured['flux'],truth,rtol=1e-5,atol=.001)
        self.assertTrue((weights>=0).all())
        np.testing.assert_allclose(weights.sum(1),1.)

    def test_all_band_color_gradients_have_independent_flux(self):
        scene=self.scene();prior=np.zeros((1,len(SCALES)));prior[0,[2,4]]=[.5,.5]
        truth={}
        for i,b in enumerate(BANDS):
            bank=dictionary(scene,b,torch.eye(2)[None])
            weights=np.zeros((1,len(SCALES)));weights[0,[2,4]]=[.1+.08*i,.9-.08*i]
            flux=2000.+100*i;truth[b]=flux
            image=np.einsum('pnk,nk->p',bank,weights)*flux+3
            scene['bands'][b]['image']=torch.tensor(image.reshape(61,61),dtype=torch.float32)
        fits=fit_multiband(scene,prior)
        for b in BANDS:
            self.assertLess(abs(fits[b]['flux'][0]/truth[b]-1),.005,b)
        fractions=[fits[b]['weights'][0,2] for b in BANDS]
        self.assertGreater(np.ptp(fractions),.4)

    def test_independent_exponential_renderer_converges(self):
        args=((71,71),np.array([35.23,34.87]),np.array([[3.,.4],[.4,2.]]),1.5)
        coarse=exponential_profile(*args,oversample=7)
        fine=exponential_profile(*args,oversample=14)
        self.assertAlmostEqual(float(coarse.sum()),1.,places=6)
        self.assertLess(np.abs(coarse-fine).sum(),.002)

if __name__=='__main__':unittest.main()
