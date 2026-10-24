import pickle
import unittest

from dependency_models import (
    CommunicationDependencyModel,
    DependencyModels,
    WaterDependencyModel,
)


class DependencyModelSerializationTests(unittest.TestCase):
    def test_pickle_uses_stable_importable_module(self):
        model = DependencyModels(
            version="test",
            water=WaterDependencyModel(("hospital/w",), (("grid/w",),), ("grid/w",)),
            communication=CommunicationDependencyModel(
                ("hospital/c",),
                (("tower/c",),),
                (("tower/c", "grid/c"),),
                ("grid/c",),
            ),
        )
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored, model)
        self.assertEqual(type(restored).__module__, "dependency_models")


if __name__ == "__main__":
    unittest.main()
