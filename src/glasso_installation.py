# src/glasso_installation.py
from rpy2.robjects import r
from rpy2.robjects.packages import importr, isinstalled

def check_and_install_glasso():
    """
    Check if glasso is installed and install it if necessary.
    Returns True if glasso is available after check/installation, False otherwise.
    """
    try:
        # Suppress R warnings and messages during package check
        r('''
        suppressWarnings(suppressMessages({
            if (requireNamespace("glasso", quietly = TRUE)) {
                TRUE
            } else {
                FALSE
            }
        }))
        ''')
        
        if isinstalled('glasso'):
            print("glasso package is already installed")
            return True
        
        print("Installing glasso package...")
        
        # Set download method to wget and install package with suppressed warnings
        r('''
        suppressWarnings(suppressMessages({
            options(download.file.method="wget")
            install.packages("glasso", repos="https://cloud.r-project.org/", quiet=TRUE)
        }))
        ''')
        
        # Verify installation quietly
        if r('''suppressWarnings(suppressMessages(requireNamespace("glasso", quietly = TRUE)))''')[0]:
            print("glasso package successfully installed")
            return True
        else:
            print("Failed to install glasso package")
            return False
            
    except Exception as e:
        print(f"Error during glasso installation: {str(e)}")
        return False