--[[ Models factory ]]

local models = {}

function models.ffnn(inDim, outDim, ...)
  local hidden = {...}
  local function addLayer(net, inD, outD)
    net:add(nn.Linear(inD,outD)):add(nn.Tanh())
  end

  local net = nn.Sequential()
  addLayer(net, inDim, inDim) -- add input layer

  -- add hidden layers
  local hidden = {...}
  for i=1,#hidden do
    addLayer(net, hidden[i-1] or inDim, hidden[i])
  end

  -- use sidmoid instead of tanh to output a probability
  net:add(nn.Linear(hidden[#hidden] or inDim, outDim)):add(nn.Sigmoid())

  return net
end

return models
