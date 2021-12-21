import os
class PrintingUtils:

    def pretty_print(title:str):
        # TODO: add windows support 
        os.system('clear')

        # print title
        print( "####" + "#" * max(len(title), 10) )
        print(f"### {title}")
        print( "####" + "#" * max(len(title), 10) )