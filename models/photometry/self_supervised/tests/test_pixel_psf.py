import unittest
import numpy as np
from scipy.special import erf
from models.photometry.self_supervised.pixel_psf import convolved_profile,gaussian_kernel


class PixelPSFTests(unittest.TestCase):
    def test_integer_point_source_preserves_asymmetric_stamp(self):
        k=np.zeros((9,9));k[4,4]=.7;k[3,5]=.3
        im=convolved_profile((41,41),(20,20),np.zeros((2,2)),k)
        np.testing.assert_allclose(im[16:25,16:25],k,atol=1e-12)
        self.assertAlmostEqual(im.sum(),1.,places=10)

    def test_gaussian_convolution_includes_pixel_response_once(self):
        k=gaussian_kernel((31,31),1.6)
        im=convolved_profile((61,61),(30,30),np.eye(2)*2.1**2,k)
        sigma=np.hypot(1.6,2.1);x=np.arange(61)-30
        a=.5*(erf((x+.5)/(np.sqrt(2)*sigma))-erf((x-.5)/(np.sqrt(2)*sigma)))
        np.testing.assert_allclose(im,np.outer(a,a),atol=1e-9)

    def test_cropped_flux_not_renormalized_and_padding_converges(self):
        k=gaussian_kernel((31,31),2.)
        a=convolved_profile((41,41),(0,20),np.eye(2)*4,k,kind='exponential')
        b=convolved_profile((41,41),(0,20),np.eye(2)*4,k,kind='exponential',padding=160)
        self.assertLess(a.sum(),.7);self.assertGreater(a.sum(),.5)
        np.testing.assert_allclose(a,b,atol=1e-8)

if __name__=='__main__':unittest.main()
