import re
import numpy as np
import pandas as pd


def parse_dti(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    
    # Handle ranges like "20%-<30%" → midpoint
    if '-' in val and '%' in val:
        nums = re.findall(r"\d+", val)
        if len(nums) == 2:
            return (int(nums[0]) + int(nums[1])) / 2
    
    # Handle less than case "<20%" → 20
    if val.startswith('<') and '%' in val:
        nums = re.findall(r"\d+", val)
        return int(nums[0])
    
    # Handle greater than case ">60%" → 60
    if val.startswith('>') and '%' in val:
        nums = re.findall(r"\d+", val)
        return int(nums[0])
    
    # Handle single percent values like "50%-60%" already covered above
    
    # Handle plain number strings (like "36", "47", etc.)
    if val.isdigit():
        return int(val)
    
    # Handle percentages like "43%" → 43
    if '%' in val:
        nums = re.findall(r"\d+", val)
        if nums:
            return int(nums[0])
    
    return np.nan