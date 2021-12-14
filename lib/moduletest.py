import unittest

class BetterUnittest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def betterRun(title:str):
        # print title
        print( "####" + "#" * max(len(title), 10) )
        print(f"### {title}")
        print( "####" + "#" * max(len(title), 10) )

        # run
        unittest.main()