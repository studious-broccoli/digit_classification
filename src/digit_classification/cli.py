import typer
from digit_classification import data, train as training_module, evaluate as evaluation_module, predict as predict_module

app = typer.Typer(
    name="digit-classifier",
    help="Digit classification CLI: download, train, evaluate, predict"
)


# Download command
@app.command()
def download_data(
    data_dir: str = typer.Option(..., "--data-dir", help="Directory to download data to")
) -> None:
    typer.echo(f"Downloading data to {data_dir}...")
    data.download_mnist(data_dir)
    typer.echo("Download complete.")


# Train command
@app.command()
def train(
    data_dir: str = typer.Option(..., "--data-dir", help="Path to training data"),
    output_dir: str = typer.Option(..., "--output-dir", help="Directory to save outputs"),
    epochs: int = typer.Option(20, "--epochs", help="Number of training epochs")
) -> None:
    typer.echo(f"Starting training for {epochs} epochs...")
    training_module.train_model(data_dir=data_dir, output_dir=output_dir, epochs=epochs)
    typer.echo("Training complete.")


# Evaluate command
@app.command()
def evaluate(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path", help="Path to model checkpoint"),
    data_dir: str = typer.Option(..., "--data-dir", help="Path to evaluation data")
) -> None:
    typer.echo(f"Evaluating model from checkpoint {checkpoint_path}...")
    evaluation_module.evaluate_model(checkpoint_path=checkpoint_path, data_dir=data_dir)
    typer.echo("Evaluation complete.")


# Predict command
@app.command()
def predict(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path", help="Path to model checkpoint"),
    input_path: str = typer.Option(..., "--input-path", help="Path to input image or data")
) -> None:
    typer.echo(f"Making prediction on input from {input_path}...")
    try:
        result = predict_module.predict_from_checkpoint(checkpoint_path=checkpoint_path, input_path=input_path)
        typer.echo(f"Prediction result: {result}")
    except Exception as e:
        typer.echo(f"[ERROR] Prediction failed: {e}", err=True)


def main():
    app()


if __name__ == "__main__":
    main()