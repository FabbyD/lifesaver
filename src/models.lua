--[[ Models factory ]]

local models = {}

function models.ffnn(loss, inDim, outDim, ...)
  -- input layer
  local par = nn.ParallelTable()
    :add(nn.Linear(inDim,inDim))
    :add(nn.Linear(inDim,inDim))
  local net = nn.Sequential()
    :add(par)
    :add(nn.CAddTable())
    :add(nn.Tanh())

  -- add hidden layers
  local function addLayer(net, inD, outD)
    net:add(nn.Linear(inD,outD)):add(nn.Tanh())
  end
  local hidden = {...}
  for i=1,#hidden do
    addLayer(net, hidden[i-1] or inDim, hidden[i])
  end

  net:add(nn.Linear(hidden[#hidden] or inDim, outDim))
  
  -- output layer depends on loss used
  if loss == 'Margin' then
    net:add(nn.Tanh())
  else
    net:add(nn.Sigmoid())
  end

  return net
end

function models.cnn(nInputDim, nOutputDim, ...)
  local net = nn.Sequential()
    :add(nn.SpatialConvolution(nInputDim))
end

return models
