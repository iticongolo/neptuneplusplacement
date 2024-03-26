from core import Application


class Node:
    def __init__(self, app: Application, hrz=0, controller=None,  monitoring=None, name="None", generator=None, nrt=None, total_rt=0,
                 subtotal_weight=0.0, total_weight=1.0, local_sla=0.0, parallel_f=None, sequential_f=None):
        self.app = app
        self.sla = self.app.sla
        self.horizon = hrz
        self.controller = controller
        self.monitoring = monitoring
        self.name = name
        self.generator = generator
        self.total_rt = total_rt
        self.nrt = nrt
        self.subtotal_weight = subtotal_weight
        self.total_weight = total_weight
        self.local_sla = local_sla
        self.parallel_f = parallel_f  # list of lists of parallel functions
        self.sequential_f = sequential_f  # list of sequential functions