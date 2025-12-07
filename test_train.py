from train import train_and_log

def test_model_quality():
    r2 = train_and_log()
    # simple quality gate: R2 must be > 0.95
    assert r2 > 0.95, f"R2 too low: {r2}"
