# tf-nmt

Program Structure:
main -> model_builder -> models + data_pipeline (iterator) -> utils (if needed to download data)

Errors:
Keep getting logits/labels mismatch in a non-deterministic way. Sometimes it will immediately throw the error, then running the program a second time will throw the error much later (or seemingly not at all).

To Test Repo:
You will need to change the hparam SAVED_MODEL_DIRECTORY to a folder to save ckpts.
Cloned repo will download all necessary dependencies, so just run main.py.
