# %% Script to contain the functions that can be used to generate the minutely text summary of the forecast data for Pirate Weather


def calculate_text(precipIntensity, prepIntensityUnit, precipType):
    """
    Calculates the textual precipitation text for the week.

    Parameters:
    - avgIntensity (float): The average precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity
    - precipType (str): The type of precipitation for the week

    Returns:
    - cSummary (str): The textual summary of conditions (very-light-rain, medium-sleet, etc.).
    """
    possibleText = ""

    if precipIntensity < (0.02 * prepIntensityUnit):
        possibleText = "possible-"

    if precipType == "rain":
        if precipIntensity < (0.4 * prepIntensityUnit):
            cSummary = possibleText + "very-light-rain"
        elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
            2.5 * prepIntensityUnit
        ):
            cSummary = "light-rain"
        elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
            10 * prepIntensityUnit
        ):
            cSummary = "medium-rain"
        else:
            cSummary = "heavy-rain"
    elif precipType == "snow":
        if precipIntensity < (0.13 * prepIntensityUnit):
            cSummary = possibleText + "very-light-snow"
        elif precipIntensity >= (0.13 * prepIntensityUnit) and precipIntensity < (
            0.83 * prepIntensityUnit
        ):
            cSummary = "light-snow"
        elif precipIntensity >= (0.83 * prepIntensityUnit) and precipIntensity < (
            3.33 * prepIntensityUnit
        ):
            cSummary = "medium-snow"
        else:
            cSummary = "heavy-snow"
    elif precipType == "sleet":
        if precipIntensity < (0.4 * prepIntensityUnit):
            cSummary = possibleText + "very-light-sleet"
        elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
            2.5 * prepIntensityUnit
        ):
            cSummary = "light-sleet"
        elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
            10 * prepIntensityUnit
        ):
            cSummary = "medium-sleet"
        else:
            cSummary = "heavy-sleet"
    else:
        # Because soemtimes there's precipitation not no type use a generic precipitation summary
        if precipIntensity < (0.4 * prepIntensityUnit):
            cSummary = possibleText + "very-light-precipitation"
        elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
            2.5 * prepIntensityUnit
        ):
            cSummary = "light-precipitation"
        elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
            10 * prepIntensityUnit
        ):
            cSummary = "medium-precipitation"
        else:
            cSummary = "heavy-precipitation"

    return cSummary


def minutely_summary(precipStart1, precipEnd1, precipStart2, text):
    """
    Calculates the textual minutely summary

    Parameters:
    - precipStart1 (int): The time the first precipitation starts
    - precipEnd1 (int): The time when the first precipitation ends
    - precipStart2 (int): The time when the precipitation starts again
    - text (str): The textual representation of the precipitation

    Returns:
    - cText (arr): The precipitation summary for the hour.
    """

    # If the current precipitation stops before the end of the hour
    if precipStart1 == 0 and precipEnd1 < 60 and precipStart2 == -1:
        cText = [
            "sentence",
            [
                "stopping-in",
                text,
                ["minutes", precipEnd1],
            ],
        ]
    # If the current precipitation doesn't stop before the end of the hour
    elif precipStart1 == 0 and precipEnd1 == 60:
        cText = [
            "sentence",
            [
                "for-hour",
                text,
            ],
        ]

    # If the current precipitation stops before the hour but starts again
    elif precipStart1 == 0 and precipEnd1 < 60 and precipStart2 != -1:
        cText = [
            "sentence",
            [
                "stopping-then-starting-later",
                text,
                ["minutes", precipEnd1],
                ["minutes", precipStart2 - precipEnd1 + 1],
            ],
        ]
    # If precip starts during the hour and lasts until the end of the hour
    elif precipStart1 > 0 and precipEnd1 == 60:
        cText = [
            "sentence",
            [
                "starting-in",
                text,
                ["minutes", precipStart1],
            ],
        ]
    # If precip starts during the hour and ends before the end of the hour
    elif precipStart1 > 0 and precipEnd1 < 60:
        cText = [
            "sentence",
            [
                "starting-then-stopping-later",
                text,
                ["minutes", precipStart1],
                ["minutes", precipEnd1 - precipStart1],
            ],
        ]
    return cText


def calculate_minutely_text(minuteArr, currentText, currentIcon):
    """
    Calculates the minutely summary given an array of minutes

    Parameters:
    - minuteArr (arr): An array of the minutes
    - currentText (str/arr): The current conditions in translations format
    - currentIcon (str): The icon representing the current conditions

    Returns:
    - cText (arr): The precipitation summary for the hour.
    - cIcon (str): The icon representing the conditions for the hour.
    """

    # Variables to use in calculating the minutely summary
    cIcon = None
    cText = None
    rainStart1 = -1
    rainEnd1 = -1
    rainStart2 = -1
    snowStart1 = -1
    snowEnd1 = -1
    snowStart2 = -1
    sleetStart1 = -1
    sleetEnd1 = -1
    sleetStart2 = -1
    noneStart1 = -1
    noneEnd1 = -1
    noneStart2 = -1
    precipStart1 = -1
    precipEnd1 = -1
    precipStart2 = -1
    avgIntensity = 0
    precipMinutes = 0
    rainMaxIntensity = 0
    snowMaxIntensity = 0
    sleetMaxIntensity = 0
    noneMaxIntensity = 0
    precipIntensityUnit = 1
    first_precip = "none"

    # Loop through the minute array
    for idx, minute in enumerate(minuteArr):
        # If there is rain for the current minute in the array
        if minute[1] == "rain" and minute[0] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            avgIntensity += minute[0]
            precipMinutes += 1
            precipIntensityUnit = minute[2]

            # Set the maxiumum rain intensity
            if rainMaxIntensity == 0:
                rainMaxIntensity = minute[0]
            elif rainMaxIntensity > 0 and minute[0] > rainMaxIntensity:
                rainMaxIntensity = minute[0]

            # Set the first precip first index if not set to the current index
            if precipStart1 == -1:
                precipStart1 = idx
            # If the first precip starting and ending index is already set then set the second starting index
            elif precipStart1 != -1 and precipEnd1 != -1 and precipStart2 == -1:
                precipStart2 = idx
            # Set the first rain first index if not set to the current index
            if rainStart1 == -1:
                rainStart1 = idx
            # If the first rain starting and ending index is already set then set the second starting index
            elif rainStart1 != -1 and rainEnd1 != -1 and rainStart2 == -1:
                rainStart2 = idx
            # If there is a first snow starting index but no ending index then set that to the current index
            if snowStart1 != -1 and snowStart2 == -1 and snowEnd1 == -1:
                snowEnd1 = idx
            # If there is a first sleet starting index but no ending index then set that to the current index
            if sleetStart1 != -1 and sleetStart2 == -1 and sleetEnd1 == -1:
                sleetEnd1 = idx
            # If there is a first none starting index but no ending index then set that to the current index
            if noneStart1 != -1 and noneStart2 == -1 and noneEnd1 == -1:
                noneEnd1 = idx
        # If there is snow for the current minute in the array
        elif minute[1] == "snow" and minute[0] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            avgIntensity += minute[0]
            precipMinutes += 1
            precipIntensityUnit = minute[2]

            # Set the maxiumum snow intensity
            if snowMaxIntensity == 0:
                snowMaxIntensity = minute[0]
            elif snowMaxIntensity > 0 and minute[0] > snowMaxIntensity:
                snowMaxIntensity = minute[0]

            # Set the first precip first index if not set to the current index
            if precipStart1 == -1:
                precipStart1 = idx
            # If the first precip starting and ending index is already set then set the second starting index
            elif precipStart1 != -1 and precipEnd1 != -1 and precipStart2 == -1:
                precipStart2 = idx
            # Set the first snow first index if not set to the current index
            if snowStart1 == -1:
                snowStart1 = idx
            # If the first snow starting and ending index is already set then set the second starting index
            elif snowStart1 != -1 and snowEnd1 != -1 and snowStart2 == -1:
                snowStart2 = idx
            # If there is a first rain starting index but no ending index then set that to the current index
            if rainStart1 != -1 and rainStart2 == -1 and rainEnd1 == -1:
                rainEnd1 = idx
            # If there is a first sleet starting index but no ending index then set that to the current index
            if sleetStart1 != -1 and sleetStart2 == -1 and sleetEnd1 == -1:
                sleetEnd1 = idx
            # If there is a first none starting index but no ending index then set that to the current index
            if noneStart1 != -1 and noneStart2 == -1 and noneEnd1 == -1:
                noneEnd1 = idx
        # If there is sleet for the current minute in the array
        elif minute[1] == "sleet" and minute[0] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            avgIntensity += minute[0]
            precipMinutes += 1
            precipIntensityUnit = minute[2]

            # Set the maxiumum sleet intensity
            if sleetMaxIntensity == 0:
                sleetMaxIntensity = minute[0]
            elif sleetMaxIntensity > 0 and minute[0] > sleetMaxIntensity:
                sleetMaxIntensity = minute[0]

            # Set the first precip first index if not set to the current index
            if precipStart1 == -1:
                precipStart1 = idx
            # If the first precip starting and ending index is already set then set the second starting index
            elif precipStart1 != -1 and precipEnd1 != -1 and precipStart2 == -1:
                precipStart2 = idx
            # Set the first sleet first index if not set to the current index
            if sleetStart1 == -1:
                sleetStart1 = idx
            # If the first sleet starting and ending index is already set then set the second starting index
            elif sleetStart1 != -1 and sleetEnd1 != -1 and sleetStart2 == -1:
                sleetStart2 = idx
            # If there is a first rain starting index but no ending index then set that to the current index
            if rainStart1 != -1 and rainStart2 == -1 and rainEnd1 == -1:
                rainEnd1 = idx
            # If there is a first snow starting index but no ending index then set that to the current index
            if snowStart1 != -1 and snowStart2 == -1 and snowEnd1 == -1:
                sleetEnd1 = idx
            # If there is a first none starting index but no ending index then set that to the current index
            if noneStart1 != -1 and noneStart2 == -1 and noneEnd1 == -1:
                noneEnd1 = idx
        elif minute[1] == "none" and minute[0] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            avgIntensity += minute[0]
            precipMinutes += 1
            precipIntensityUnit = minute[2]

            # Set the none maxiumum precipitation intensity
            if noneMaxIntensity == 0:
                noneMaxIntensity = minute[0]
            elif noneMaxIntensity > 0 and minute[0] > noneMaxIntensity:
                noneMaxIntensity = minute[0]

            # Set the first precip first index if not set to the current index
            if precipStart1 == -1:
                precipStart1 = idx
            # If the first precip starting and ending index is already set then set the second starting index
            elif precipStart1 != -1 and precipEnd1 != -1 and precipStart2 == -1:
                precipStart2 = idx
            # Set the first none first index if not set to the current index
            if noneStart1 == -1:
                noneStart1 = idx
            # If the first none starting and ending index is already set then set the second starting index
            elif noneStart1 != -1 and noneEnd1 != -1 and noneStart2 == -1:
                noneStart2 = idx
            # If there is a first rain starting index but no ending index then set that to the current index
            if rainStart1 != -1 and rainStart2 == -1 and rainEnd1 == -1:
                rainEnd1 = idx
            # If there is a first snow starting index but no ending index then set that to the current index
            if snowStart1 != -1 and snowStart2 == -1 and snowEnd1 == -1:
                snowEnd1 = idx
        # If there is no precipitation for the current minute
        else:
            # If there is a first precip starting index but no ending index then set that to the current index
            if precipStart1 != -1 and precipStart2 == -1 and precipEnd1 == -1:
                precipEnd1 = idx
            # If there is a first rain starting index but no ending index then set that to the current index
            if rainStart1 != -1 and rainStart2 == -1 and rainEnd1 == -1:
                rainEnd1 = idx
            # If there is a first snow starting index but no ending index then set that to the current index
            if snowStart1 != -1 and snowStart2 == -1 and snowEnd1 == -1:
                snowEnd1 = idx
            # If there is a first sleet starting index but no ending index then set that to the current index
            if sleetStart1 != -1 and snowStart2 == -1 and snowEnd1 == -1:
                sleetEnd1 = idx
            # If there is a first none starting index but no ending index then set that to the current index
            if noneStart1 != -1 and noneStart2 == -1 and noneEnd1 == -1:
                noneEnd1 = idx

    # If there is a first starting index for rain/snow/sleet but no ending index then set it to 60
    if precipStart1 != -1 and precipEnd1 == -1:
        precipEnd1 = 60
    if rainStart1 != -1 and rainEnd1 == -1:
        rainEnd1 = 60
    if snowStart1 != -1 and snowEnd1 == -1:
        snowEnd1 = 60
    if sleetStart1 != -1 and sleetEnd1 == -1:
        sleetEnd1 = 60
    if noneStart1 != -1 and noneEnd1 == -1:
        noneEnd1 = 60

    # Calculate the average precipitaiton intensity
    if precipMinutes > 0:
        avgIntensity = avgIntensity / precipMinutes

    # Create an array of the starting times for the precipitation
    starts = []

    if sleetStart1 >= 0:
        starts.append(sleetStart1)
    if snowStart1 >= 0:
        starts.append(snowStart1)
    if rainStart1 >= 0:
        starts.append(rainStart1)
    if noneStart1 >= 0:
        starts.append(noneStart1)

    # If the array has any values check the minimum against the different precipitation start times and set that as the first precipitaion
    if starts:
        if sleetStart1 == min(starts):
            first_precip = "sleet"
        elif snowStart1 == min(starts):
            first_precip = "snow"
        elif rainStart1 == min(starts):
            first_precip = "rain"
        elif noneStart1 == min(starts):
            first_precip = "none"

    # If there are more than two precipitation types used the mixed text
    if len(starts) > 2:
        text = "mixed-precipitation"
        cIcon = "sleet"
    # If there is two precipitation types for the hour
    elif len(starts) == 2:
        # Calculate the maximum intensity based on the precipitation types
        maxIntensity = max(
            rainMaxIntensity, snowMaxIntensity, sleetMaxIntensity, noneMaxIntensity
        )
        # If the first type is sleet show that as the minutely summary text and set the icon to sleet
        if first_precip == "sleet":
            text = calculate_text(maxIntensity, precipIntensityUnit, "sleet")
            cIcon = "sleet"
        # If the first type is snow show that as the minutely summary text and set the icon to snow
        elif first_precip == "snow":
            text = calculate_text(maxIntensity, precipIntensityUnit, "snow")
            cIcon = "snow"
        # If the first type is rain show that as the minutely summary text and set the icon to rain
        elif first_precip == "rain":
            text = calculate_text(maxIntensity, precipIntensityUnit, "rain")
            cIcon = "rain"
        # If the first type has no type that as the minutely summary text and set the icon to rain
        else:
            text = calculate_text(maxIntensity, precipIntensityUnit, "none")
            cIcon = "rain"

    # If there is no precipitation then set the minutely summary/icon to the current icon/summary
    if (
        rainStart1
        == rainEnd1
        == rainStart2
        == snowStart1
        == snowEnd1
        == snowStart2
        == sleetStart1
        == sleetEnd1
        == sleetStart2
        == noneStart1
        == noneEnd1
        == noneStart2
        == -1
    ):
        cText = ["sentence", ["for-hour", currentText]]
        cIcon = currentIcon
    # If there is more than one precipitation for the hour
    elif len(starts) > 1:
        # Calculate the text using the start/end times for precipitation as a whole instead of the individual precipitation
        cText = minutely_summary(
            precipStart1,
            precipEnd1,
            precipStart2,
            text,
        )
    # If there if the only one precipitation is sleet
    elif sleetStart1 != -1:
        cText = minutely_summary(
            sleetStart1,
            sleetEnd1,
            sleetStart2,
            calculate_text(sleetMaxIntensity, precipIntensityUnit, "sleet"),
        )
        cIcon = "sleet"
    # If there if the only one precipitation is snow
    elif snowStart1 != -1:
        cText = minutely_summary(
            snowStart1,
            snowEnd1,
            snowStart2,
            calculate_text(snowMaxIntensity, precipIntensityUnit, "snow"),
        )
        cIcon = "snow"
    # If there if the only one precipitation is rain
    elif rainStart1 != -1:
        cText = minutely_summary(
            rainStart1,
            rainEnd1,
            rainStart2,
            calculate_text(rainMaxIntensity, precipIntensityUnit, "rain"),
        )
        cIcon = "rain"
    # If there if the only one precipitation has any other type
    else:
        cText = minutely_summary(
            noneStart1,
            noneEnd1,
            noneStart2,
            calculate_text(noneMaxIntensity, precipIntensityUnit, "none"),
        )
        cIcon = "rain"

    return cText, cIcon
