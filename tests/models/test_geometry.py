import unittest

import pandas as pd
import numpy as np

from ananke.models.geometry import (
    Vectors2D,
    Vectors3D,
    LocatedObjects,
    OrientedLocatedObjects,
)


class Vectors2DTestCase(unittest.TestCase):
    def test_scaling(self):
        vectors = Vectors2D(
            df=pd.DataFrame(
                {
                    'x': [0, 3, 0],
                    'y': [3, 0, 0]
                }
            )
        )

        vectors.scale_to_length(2)

        np.testing.assert_array_equal(
            vectors.to_numpy(), np.array(
                [
                    [0, 2],
                    [2, 0],
                    [0, 0]
                ]
            )
        )

    def test_phi(self):
        vectors = Vectors2D(
            df=pd.DataFrame(
                {
                    'x': [0, 1, -1, 0],
                    'y': [1, 0, 0, -1]
                }
            )
        )
        phi = vectors.phi.to_numpy()
        self.assertTrue(np.all(np.isclose(phi, [np.pi / 2, 0, np.pi, -1 / 2 * np.pi])))

    def test_norm(self):
        vectors = Vectors2D(
            df=pd.DataFrame(
                {
                    'x': [0, 2, 0, 2],
                    'y': [3, 0, 0, 3]
                }
            )
        )

        self.assertTrue(
            vectors.norm.equals(pd.Series([3, 2, 0, np.sqrt(2 ** 2 + 3 ** 2)]))
        )

    def test_polar_conversion(self):
        polar_df = pd.DataFrame(
            {
                'norm': [2, 1],
                'phi': [np.pi / 2, 2 * np.pi]
            }
        )

        vectors = Vectors2D.from_polar(polar_df)

        self.assertTrue(np.all(np.isclose(vectors.to_numpy(), [[0, 2], [1, 0]])))

    def test_from_numpy(self):
        array = [[0, 2], [1, 0]]
        vectors = Vectors2D.from_numpy(numpy_array=np.array(array))

        assert vectors == Vectors2D(
            df=pd.DataFrame(
                {
                    'x': [0, 1],
                    'y': [2, 0]
                }
            )
        )


class Vectors3DTestCase(unittest.TestCase):
    def test_non_polar(self):
        with self.assertRaises(AttributeError):
            polar_df = pd.DataFrame(
                {
                    'norm': [2, 1],
                    'phi': [np.pi / 2, 2 * np.pi]
                }
            )
            Vectors3D.from_polar(polar_df)

    def test_from_spherical(self):
        spherical_df = pd.DataFrame(
            {
                'norm': [2, 1, 1, 2],
                'phi': [np.pi / 2, np.pi, 0, 0],
                'theta': [np.pi / 2, np.pi / 2, 0, np.pi]
            }
        )
        vectors = Vectors3D.from_spherical(spherical_df)

        self.assertTrue(
            np.all(
                np.isclose(
                    vectors.to_numpy(), np.array(
                        [
                            [0, -1, 0, 0],
                            [2, 0, 0, 0],
                            [0, 0, 1, -2]
                        ]
                    ).T
                )
            )
        )

    def test_theta(self):
        theta = [np.pi / 2, np.pi / 2, 0, np.pi]
        vectors = Vectors3D.from_spherical(pd.DataFrame(
            {
                'norm': [2, 1, 1, 2],
                'phi': [np.pi / 2, np.pi, 0, 0],
                'theta': theta
            }
        ))
        self.assertTrue(np.all(np.isclose(vectors.theta, theta)))

    def test_phi(self):
        theta = [np.pi / 2, np.pi / 2, 0, np.pi]
        phi = [np.pi / -2, np.pi, 0, 0]
        vectors = Vectors3D.from_spherical(pd.DataFrame(
            {
                'norm': [2, 1, 1, 2],
                'phi': phi,
                'theta': theta
            }
        ))
        self.assertTrue(np.all(np.isclose(vectors.phi, phi)))


    def test_from_numpy(self):
        array = [[0, 2, 0], [1, 0, -1]]
        vectors = Vectors3D.from_numpy(numpy_array=np.array(array))

        assert vectors == Vectors3D(
            df=pd.DataFrame(
                {
                    'x': [0, 1],
                    'y': [2, 0],
                    'z': [0, -1]
                }
            )
        )

    def test_prefixing(self):
        vectors = Vectors3D(
            df=pd.DataFrame(
                {
                    'x': [0, 1],
                    'y': [2, 0],
                    'z': [0, -1]
                }
            )
        )

        self.assertTrue(
            vectors.get_df_with_prefix('test_').equals(
                pd.DataFrame(
                    {
                        'test_x': [0., 1.],
                        'test_y': [2., 0.],
                        'test_z': [0., -1.]
                    }
                )
            )
        )

    def test_unprefixing(self):
        vectors = Vectors3D(
            df=pd.DataFrame(
                {
                    'x': [0, 1],
                    'y': [2, 0],
                    'z': [0, -1]
                }
            )
        )

        df = pd.DataFrame(
            {
                'test_x': [0., 1.],
                'test_y': [2., 0.],
                'test_z': [0., -1.]
            }
        )

        assert vectors == Vectors3D.from_df(df, prefix='test_')


class LocatedObjectTestCase(unittest.TestCase):
    def test_locations_property(self):
        locations = LocatedObjects(
            df=pd.DataFrame(
                {
                    'location_x': [0., 1.],
                    'location_y': [2., 0.],
                    'location_z': [0., -1.]
                }
            )
        )

        vectors = Vectors3D(
            df=pd.DataFrame(
                {
                    'x': [0, 1],
                    'y': [2, 0],
                    'z': [0, -1]
                }
            )
        )

        assert vectors == locations.locations


class OrientedLocatedObjectsTestCase(unittest.TestCase):
    def test_orientation_property(self):
        orientated_locations = OrientedLocatedObjects(
            df=pd.DataFrame(
                {
                    'location_x': [0., 1.],
                    'location_y': [2., 0.],
                    'location_z': [0., -1.],
                    'orientation_x': [2., 0.],
                    'orientation_y': [0., 3.],
                    'orientation_z': [1., 1.]
                }
            )
        )

        vectors = Vectors3D(
            df=pd.DataFrame(
                {
                    'x': [2, 0],
                    'y': [0, 3],
                    'z': [1, 1]
                }
            )
        )

        assert vectors == orientated_locations.orientations


if __name__ == '__main__':
    unittest.main()
