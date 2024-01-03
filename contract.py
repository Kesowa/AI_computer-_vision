# Code should be platform-independent
# Do not assume paths will remain the same ("C:\some\file" on Win, "/some/file" on Unix)
class Model:
    def __init__(self, model_path="checkpoint.pth"):
        # Load model, compile it for optimizations, etc over here
        # Runs once throughout the application lifecycle
        # Expensive, hence the initialized class will be re-used for multiple inferences

    def infer(self, data_path="raster.tif"):
        # Run model on provided data
        # Does not stores any information to disk or model instance
        # Runs multiple times, once for each inference
        # Returns a json-like object, which will then be sent over HTTP
        # Does NOT return a file path
        return geojson_object
