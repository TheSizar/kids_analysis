"""
This script provides functions to style pandas DataFrames to look like Excel tables.
You can import this in your Jupyter notebook and use the functions to style your dataframes.
"""

import pandas as pd
from IPython.display import display, HTML
import sys
from io import StringIO

# The core styling function
def style_dataframe(df):
    """
    Apply Excel-like styling to a pandas DataFrame
    
    Args:
        df: pandas DataFrame to style
    
    Returns:
        Styled DataFrame
    """
    # Create the styled dataframe
    styled = df.style.set_properties(**{
        'background-color': '#f5f5f5',
        'color': 'black',
        'border-color': '#999999',
        'border-style': 'solid',
        'border-width': '1px',
        'text-align': 'left'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#4472C4'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('border', '1px solid #999999'),
            ('text-align', 'center'),
            ('padding', '5px')
        ]},
        {'selector': 'td', 'props': [
            ('padding', '5px'),
            ('border', '1px solid #999999')
        ]},
        {'selector': 'tr:nth-of-type(odd)', 'props': [
            ('background-color', '#ffffff')
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#e6f2ff')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#ffffcc')
        ]}
    ])
    
    # Try to hide index if the method exists, otherwise continue without it
    try:
        styled = styled.hide_index()
    except AttributeError:
        # For older pandas versions that don't have hide_index()
        pass
        
    return styled

# Simple function to override print for pandas DataFrames
def enable_styled_print():
    """
    Override the built-in print function to automatically style pandas DataFrames.
    This allows you to continue using print(df) syntax while getting styled output.
    """
    from IPython import get_ipython
    
    # Store the original print function
    original_print = print
    
    # Define our custom print function
    def styled_print(*args, **kwargs):
        # Check if we're in a Jupyter notebook
        ipython = get_ipython()
        if ipython is None or 'IPKernelApp' not in ipython.config:
            # Not in a notebook, use regular print
            return original_print(*args, **kwargs)
        
        # Process each argument
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # If it's a DataFrame, display it with styling
                display(style_dataframe(arg))
            else:
                # For other types, use the original print
                original_print(arg, **kwargs)
    
    # Replace the built-in print function
    import builtins
    builtins.print = styled_print
    
    print("Excel-like styling enabled for DataFrames with print()!")
    print("Your existing print(df) statements will now show styled tables.")
    print("To disable, call disable_styled_print()")

def disable_styled_print():
    """
    Restore the original print function.
    """
    import builtins
    builtins.print = __builtins__.print
    print("Print function restored to default behavior.")

# Keep these other functions for flexibility
def display_styled(df):
    """
    Display a dataframe with Excel-like styling
    
    Args:
        df: pandas DataFrame to display with styling
    """
    styled_df = style_dataframe(df)
    display(styled_df)

def add_excel_styling():
    """
    Add a function to the IPython environment that makes it easy to display
    Excel-styled dataframes.
    
    Usage in notebook after importing this module:
    1. Run add_excel_styling()
    2. Then use excel_df(your_dataframe) to display styled dataframes
    """
    from IPython import get_ipython
    
    # Define the function that will be available in the notebook
    def excel_df(df):
        """Display a dataframe with Excel-like styling"""
        styled = style_dataframe(df)
        display(styled)
    
    # Add it to the IPython namespace
    ipython = get_ipython()
    if ipython is not None:
        ipython.user_ns['excel_df'] = excel_df
        print("Excel styling function added! Use excel_df(your_dataframe) to display styled dataframes.")
    else:
        print("Not running in IPython environment. Use display_styled(your_dataframe) instead.")

def enable_auto_styling():
    """
    Enable automatic styling of all pandas DataFrames in the notebook.
    Call this function at the beginning of your notebook.
    
    Note: This approach may not work in all Jupyter environments.
    If it doesn't work, use add_excel_styling() instead.
    """
    try:
        old_repr = pd.DataFrame._repr_html_
        
        def new_repr(self):
            return style_dataframe(self)._repr_html_()
        
        pd.DataFrame._repr_html_ = new_repr
        print("Excel-like styling enabled for all DataFrames!")
        print("If you don't see styled dataframes, try using add_excel_styling() instead.")
    except Exception as e:
        print(f"Could not enable auto-styling: {e}")
        print("Please use add_excel_styling() instead.")
    
def disable_auto_styling():
    """
    Disable automatic styling and return to default pandas display.
    """
    try:
        pd.DataFrame._repr_html_ = pd.DataFrame._repr_html_
        print("DataFrame styling reset to default.")
    except Exception as e:
        print(f"Could not disable auto-styling: {e}")

# Example usage:
# 1. Import this module in your notebook:
#    from style_dataframes import enable_styled_print
#
# 2. Enable styled printing:
#    enable_styled_print()
#
# 3. Use your existing print statements:
#    print(df.head())  # Will now show styled output