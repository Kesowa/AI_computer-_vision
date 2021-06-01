from deepforest import deepforest
from deepforest import get_data


MODEL_SAVE_PATH = r"/home/ricky/Desktop/Kesowa/Tree Detection/DeepForest"

test_model = deepforest.deepforest()

# Example run with short training
test_model.config["epochs"] = 20
test_model.config["save-snapshot"] = False
test_model.config["steps"] = 1
test_model.config["batch_size"] = 1

annotations_file = get_data("train.csv")
test_model.train(annotations=annotations_file, input_type="fit_generator")

# saving the model
os.chdir(MODEL_SAVE_PATH)
test_model.model.save("tree_detector.h5")