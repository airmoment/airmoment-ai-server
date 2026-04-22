def decide(predicted_drop_amount):
    if predicted_drop_amount >= 15000:
        return "WAIT"
    return "BUY"