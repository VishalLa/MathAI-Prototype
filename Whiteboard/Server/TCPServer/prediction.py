from CNN_Model.Utils.pre_process import predict_chars

class PredictionHandler:
    def __init__(self, main_window):
        self.main = main_window

    def predict(self):
        selected_area = self.main.scene.get_selected_region()
        if selected_area.size == 0:
            print("No area selected")
            return

        chars = predict_chars(selected_area)
        print(f"Predicted characters: {chars}")
