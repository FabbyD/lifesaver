--[[ Models factory ]]

local models = {}

function models.ffnn(inDim, outDim, ...)
  -- input layer
  local par = nn.ParallelTable()
    :add(nn.Linear(inDim,inDim))
    :add(nn.Linear(inDim,inDim))
  local net = nn.Sequential()
    :add(par)
    :add(nn.JoinTable(1))
    :add(nn.Sigmoid())

  -- add hidden layers
  local function addLayer(net, inD, outD)
    net:add(nn.Linear(inD,outD)):add(nn.Sigmoid())
  end
  local hidden = {...}
  for i=1,#hidden do
    addLayer(net, hidden[i-1] or 2*inDim, hidden[i])
  end

  -- use sidmoid instead of tanh to output a probability
  net:add(nn.Linear(hidden[#hidden] or 2*inDim, outDim)):add(nn.Sigmoid())

  return net
end

return models
