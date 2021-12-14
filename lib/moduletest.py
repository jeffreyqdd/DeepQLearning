import unittest
import os

class BetterUnittest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def betterRun(title:str):
        # TODO: add windows support 
        os.system('clear')

        # print title
        print( "####" + "#" * max(len(title), 10) )
        print(f"### {title}")
        print( "####" + "#" * max(len(title), 10) )

        # run
        unittest.main()