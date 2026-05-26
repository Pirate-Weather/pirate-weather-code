from datetime import datetime, timedelta

import pytz


def get_period(hour):
    if 4 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 21:
        return "evening"
    return "night"


def run_dst(start_date, label):
    print(f"\n--- {label} ---")
    tz = pytz.timezone("US/Eastern")
    start_dt = tz.localize(datetime.strptime(start_date, "%Y-%m-%d"))

    # Generate 72 hours using UTC offsets to handle DST transitions correctly
    times_utc = [start_dt.astimezone(pytz.utc) + timedelta(hours=i) for i in range(72)]
    times_local = [t.astimezone(tz) for t in times_utc]

    slices = [(4, 17), (17, 28), (28, 41), (41, 52)]
    for start, end in slices:
        subset = times_local[start:end]
        periods = sorted(list(set(get_period(t.hour) for t in subset)))
        times_str = f"{subset[0].strftime('%H:%M')} ({subset[0].strftime('%Z')}) to {subset[-1].strftime('%H:%M')} ({subset[-1].strftime('%Z')})"
        print(
            f"Slice [{start}:{end}]: {times_str} | Periods ({len(periods)}): {', '.join(periods)}"
        )


run_dst("2024-03-10", "Spring Forward (2024-03-10)")
run_dst("2024-11-03", "Fall Back (2024-11-03)")
