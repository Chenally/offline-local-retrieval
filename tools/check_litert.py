from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "week1_test.tflite"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LiteRT model was not found: {MODEL_PATH}"
        )

    print(f"Loading model: {MODEL_PATH}")

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        raise RuntimeError("The model has no input tensors.")

    if not output_details:
        raise RuntimeError("The model has no output tensors.")

    for input_detail in input_details:
        input_shape = input_detail["shape"]
        input_dtype = input_detail["dtype"]
        input_index = input_detail["index"]

        input_data = np.zeros(
            input_shape,
            dtype=input_dtype,
        )

        interpreter.set_tensor(
            input_index,
            input_data,
        )

        print(
            "Input tensor:",
            input_detail["name"],
            "shape =",
            input_shape,
            "dtype =",
            input_dtype,
        )

    print("Running inference...")
    interpreter.invoke()

    for output_detail in output_details:
        output_data = interpreter.get_tensor(
            output_detail["index"],
        )

        if output_data.size == 0:
            raise RuntimeError(
                f"Output tensor is empty: {output_detail['name']}"
            )

        print(
            "Output tensor:",
            output_detail["name"],
            "shape =",
            output_data.shape,
            "dtype =",
            output_data.dtype,
        )

    print("LiteRT inference check passed.")


if __name__ == "__main__":
    main()
