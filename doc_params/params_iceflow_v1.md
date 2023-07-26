usage: PAR.py [-h] [--init_strflowctrl INIT_STRFLOWCTRL] [--emulator EMULATOR]
              [--init_slidingco INIT_SLIDINGCO]
              [--init_arrhenius INIT_ARRHENIUS]
              [--multiple_window_size MULTIPLE_WINDOW_SIZE]
              [--force_max_velbar FORCE_MAX_VELBAR]

optional arguments:
  -h, --help            show this help message and exit
  --init_strflowctrl INIT_STRFLOWCTRL
                        Initial strflowctrl (default 78)
  --emulator EMULATOR   Directory path of the deep-learning ice flow model,
                        create a new if empty string
  --init_slidingco INIT_SLIDINGCO
                        Initial sliding coeeficient slidingco (default: 0)
  --init_arrhenius INIT_ARRHENIUS
                        Initial arrhenius factor arrhenuis (default: 78)
  --multiple_window_size MULTIPLE_WINDOW_SIZE
                        If a U-net, this force window size a multiple of 2**N
                        (default: 0)
  --force_max_velbar FORCE_MAX_VELBAR
                        This permits to artif. upper-bound velocities, active
                        if > 0 (default: 0)
