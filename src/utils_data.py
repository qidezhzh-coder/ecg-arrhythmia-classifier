import os

def get_valid_records(data_path, required_exts=(".hea", ".dat", ".atr")):
    all_files = os.listdir(data_path)

    candidates = sorted(
        {os.path.splitext(f)[0] for f in all_files if f.endswith(".hea")},
        key=lambda x: int(x) if x.isdigit() else x
    )

    valid_records = []
    incomplete = {}

    for rec in candidates:
        missing = [
            ext for ext in required_exts
            if not os.path.exists(os.path.join(data_path, rec + ext))
        ]
        if missing:
            incomplete[rec] = missing
        else:
            valid_records.append(rec)

    return valid_records, incomplete