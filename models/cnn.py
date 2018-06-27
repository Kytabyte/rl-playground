class Conv2dNet(nn.Module):
    def __init__(self, input_size, in_channel, num_output):
        super(Conv2dNet, self).__init__()
        
        self._input_width, self._input_height = input_size
        # this could be `in_channel` for conv models or `num_input` for linear models
        self._last_input = in_channel
        self._num_output = num_output
        
        self._conv = nn.Sequential()
        self._num_conv = 0
        self._fc = nn.Sequential()
        self._num_fc = 0
        
        self._done = False
        
        
    def add_conv(self, out_channel, kernel_size, activation=nn.ReLU, **kwargs):
        if self._done:
            raise Exception('No module can be added after calling setup()')
        
        self._num_conv += 1
        self._conv.add_module('conv{}'.format(self._num_conv),
                              nn.Conv2d(self._last_input, out_channel, kernel_size, **kwargs))
        self._conv.add_module('act-conv{}'.format(self._num_conv),
                              activation())
        
        self._last_input = out_channel
        
        
    def add_fc(self, num_output, activation=nn.ReLU, **kwargs):
        if self._done:
            raise Exception('No module can be added after calling setup()')
            
        self._num_fc += 1
        self._fc.add_module('fc{}'.format(self._num_fc),
                            nn.Linear(self._last_input, num_output, **kwargs))
        if activation:
            self._fc.add_module('act-fc{}'.format(self._num_conv),
                              activation())
        
        self._last_input = num_output
            
    def add_batch_norm(self, **kwargs):
        if self._done:
            raise Exception('No module can be added after calling setup()')
            
        self._conv.add_module('batch norm{}'.format(self._num_conv),
                              nn.BatchNorm2d(self._last_input, **kwargs))
    
    def add_pool(self, kernel_size, **kwargs):
        if self._done:
            raise Exception('No module can be added after calling setup()')
            
        self._conv.add_module('pooling{}'.format(self._num_conv),
                              nn.MaxPool2d(kernel_size, **kwargs))
        
        
    def add_dropout(self, ratio=0.5, **kwargs):
        if self._done:
            raise Exception('No module can be added after calling setup()')
            
        self._fc.add_module('dropout{}'.format(self._num_fc),
                                nn.Dropout(ratio, **kwargs))
        
    def flatten(self):
        if self._done:
            raise Exception('No module can be added after calling setup()')
            
        x = torch.rand((1, 1, self._input_width, self._input_height))
        x = self._conv(x)
        x = x.reshape((x.size(0), -1)).detach()
        self._last_input = x.size(1)
        
            
    def setup(self):
        if self._num_fc == 0:
            self.flatten()
            self.add_fc(self._num_output)
            
        self._done = True
        
    def forward(self, x):
        if not self._done:
            raise('model cannot run before calling setup()')
            
        x = self._conv(x)
        x = x.reshape((x.size(0), -1))
        self._num_flat_features = x.size(1)