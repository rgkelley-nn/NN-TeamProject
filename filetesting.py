# Source - https://stackoverflow.com/a/12280052
# Posted by thclpr, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-28, License - CC BY-SA 4.0

import os
import glob

path = 'data/'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
