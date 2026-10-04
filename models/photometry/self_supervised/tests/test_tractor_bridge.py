"""Run in the isolated Tractor environment; tests integration, not sky accuracy."""
import unittest
import importlib.util
import numpy as np
HAVE_TRACTOR = importlib.util.find_spec("tractor") is not None
if HAVE_TRACTOR:
    from tractor import PointSource, RaDecPos, NanoMaggies
    from models.photometry.self_supervised.tractor_compare import image_for_band, joint_flux


@unittest.skipUnless(HAVE_TRACTOR, "Run in the isolated Tractor environment")
class TractorBridgeTest(unittest.TestCase):
    def fixture(self):
        xy=np.array([[20.,20.],[22.,20.]])
        sky=np.array([[53.,-28.],[53.-.2/3600/np.cos(np.deg2rad(-28)),-28.]])
        return dict(sky=sky,euclid_VIS__image=np.zeros((43,43)),
                    euclid_VIS__variance=np.full((43,43),.04),
                    euclid_VIS__mask=np.ones((43,43),bool),
                    euclid_VIS__positions=xy,euclid_VIS__sky_to_pixel=np.diag([-10.,10.]),
                    euclid_VIS__psf_sigma=1.5)

    def test_joint_signed_flux_sky_and_covariance(self):
        scene=self.fixture();tim=image_for_band(scene,'euclid_VIS')
        sources=[PointSource(RaDecPos(*p),NanoMaggies(VIS=1.)) for p in scene['sky']]
        columns=[]
        for source in sources:
            column=np.zeros(tim.shape);source.getModelPatch(tim).addTo(column);columns.append(column)
        truth=np.array([7.,-2.]);sky=.37
        tim.data[:]=sky+sum(f*c for f,c in zip(truth,columns))
        result=joint_flux(tim,sources,'VIS')
        np.testing.assert_allclose(result['flux'],truth,atol=1e-6)
        self.assertAlmostEqual(result['sky'],sky,places=6)
        a=np.column_stack([c.ravel() for c in columns]+[np.ones(tim.data.size)])/.2
        expected=np.sqrt(np.diag(np.linalg.inv(a.T@a))[:-1])
        np.testing.assert_allclose(result['error'],expected,rtol=1e-6)
        self.assertTrue((result['error']>1/np.sqrt(np.sum(a[:,:2]**2,axis=0))).all())

    def test_pixelized_psf_tractor_profile_convention(self):
        from tractor import ExpGalaxy, GalaxyShape
        from models.photometry.self_supervised.pixel_psf import convolved_profile, gaussian_kernel
        scene=self.fixture()
        kernel=gaussian_kernel((31,31),1.5)
        # An asymmetric wing tests actual pixelized convolution, not just width.
        kernel=.94*kernel+.06*np.roll(kernel,4,axis=1)
        scene['euclid_VIS__psf_kernels']=np.stack([kernel,kernel])
        truth=np.array([7.,3.])
        scale=1.7  # exponential radius in native pixels
        # Use Tractor's documented MoG approximation to the exponential for
        # this interface test: an exact exponential differs from that profile.
        from tractor.mixture_profiles import get_exp_mixture
        profile=get_exp_mixture()
        profiles=[sum(amp*convolved_profile((43,43),xy,cov*(scale*1.67834699)**2,kernel)
                      for amp,cov in zip(profile.amp,profile.var))
                  for xy in scene['euclid_VIS__positions']]
        scene['euclid_VIS__image']=.1+sum(f*t for f,t in zip(truth,profiles))
        tim=image_for_band(scene,'euclid_VIS')
        sources=[ExpGalaxy(RaDecPos(*p),NanoMaggies(VIS=1.),GalaxyShape(scale*.1*1.67834699,1.,0.))
                 for p in scene['sky']]
        result=joint_flux(tim,sources,'VIS')
        np.testing.assert_allclose(result['flux'],truth,rtol=1e-3)

    def test_masked_contamination_ignored(self):
        scene=self.fixture();scene['euclid_VIS__mask'][0,0]=False
        scene['euclid_VIS__image'][0,0]=np.nan
        tim=image_for_band(scene,'euclid_VIS')
        self.assertEqual(tim.getInvvar()[0,0],0)
        self.assertTrue(np.isfinite(tim.getImage()).all())

if __name__=='__main__':unittest.main()
