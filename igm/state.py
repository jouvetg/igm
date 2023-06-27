#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import logging, os 
 
class State:

    def __init__(self,params):
        """
        function build class IGM
        """

        # this serves in fact to use a dictionnary, but it looks like a class
        self.__dict__ = dict()
 
        # this will collect  statisitics of the computational times
        self.tcomp = {}
         
        # Configure the logging
        if params.logging_file=='':
            logging.basicConfig(
                                filemode="w",
                                level=getattr(logging,params.logging_level),
                                format='%(asctime)s - %(levelname)s - %(message)s'
                               )
        else:            
            logging.basicConfig(
                                filename=params.logging_file, 
                                encoding='utf-8', 
                                filemode="w",
                                level=getattr(logging,params.logging_level),
                                format='%(asctime)s - %(levelname)s - %(message)s'
                                )

        # Create a logger
        self.logger = logging.getLogger('my_logger')
                
        os.system('rm clean.sh')
        os.system('echo rm clean.sh >> clean.sh')
        os.system('echo rm igm.log >> clean.sh')
        

        