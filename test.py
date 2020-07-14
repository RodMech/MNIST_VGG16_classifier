from src.inferencer import Inferencer

if __name__ == '__main__':
    inferencer = Inferencer()
    inferencer.load_test_data()
    inferencer.perform_inference()