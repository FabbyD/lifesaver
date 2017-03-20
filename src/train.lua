require 'nn'
require 'player'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training Lifesaver - a neural Minesweeper')
cmd:text()
cmd:text('Options')
cmd:option('-size', 15, 'The size of the field')
cmd:option('-bombs', 50, 'Number of bombs')
cmd:option('-t', 0.5, 'Bomb certainty level')
cmd:option('-games', 1000, 'Number of games to play')
cmd:text()

local opts = cmd:parse(arg)

local optim = require 'optim'
local models = require 'models'

local size = opts.size
local numBombs = opts.bombs
local threshold = opts.t
local numCells = size*size

local player = Player(size, numBombs)
local net = models.ffnn(numCells, numCells, numCells*2)
local criterion = nn.BCECriterion() -- Binary Cross Entropy for Sigmoid output layer
local config = {learningRate=0.001} -- optim config

local field   = player.field
local bombs   = player.bombs
local visible = player.visible
local flags   = player.flags

print('## Architecture:')
print(net)

print('## Setup:')
print('Size :     ' .. size)
print('Bombs:     ' .. numBombs)
print('Threshold: ' .. threshold)
print('Number of games: ' .. opts.games)

local function findNthElem(t, n, elem)
  local count = 0
  for i=1,t:size(1) do
    if t[i] == elem then
      count = count + 1
      if count == n then
        return i
      end
    end
  end
end

local function field2string(field)
  local str = tostring(field)
    :gfind('(.+)\n%[')()
    :gsub('^','  ')
    :gsub('\n','\n  ')
    :gsub('(%d+%.%d)%d*', '%1') -- keep 1 digit only
  return str
end

local params, gradParams = net:getParameters()
local outfh = io.open('outputs.txt', 'w')

local function train()
  local done = false
  local sumLoss = 0
  local rounds = 0
  while not done do
    local input = visible:view(-1) -- flatten
    local invis = torch.eq(visible, player.INVIS)
    local mask = torch.cbitxor(invis,bombs:byte())
    if mask:sum() == 0 then -- everything is revealed or bombs
      done = true
    else
      rounds = rounds + 1
      local output = net:forward(input)
      outfh:write('Round ' .. rounds .. '\n')
      outfh:write(field2string(visible) .. '\n\n')
      outfh:write(field2string(output:view(size,size)) .. '\n\n')
      local target = bombs:view(-1)
      local loss = criterion:forward(output, target)
      sumLoss = sumLoss + loss
      local gradOutput = criterion:backward(output, target)
      net:backward(input, gradOutput)
      player:triggerMin(output)
    end
  end
  return sumLoss
end

local function evaluate()
  print('Evaluating!')
  for g=1,opts.evalGames do
    local done = false
    while not done do
      local input = visible:view(-1)

    end
  end
end

local wins = 0
local currWins = 0
local cumulLoss = 0
for g=1,opts.games do
  player:resetField()
  outfh:write('### Game ' .. g .. '\n')
  outfh:write('===== Field ======\n')
  outfh:write(field2string(field) .. '\n')
  outfh:write('==================\n')

  local function feval(x)
    if x ~= params then
      print('WARNING: Copying parameters')
      params:copy(x)
    end

    gradParams:zero()

    local loss = train()
    cumulLoss = cumulLoss + loss

    return loss, gradParams
  end
  optim.sgd(feval, params, config)

  if g%100==0 then
    print('game ' .. g .. ' loss: ' .. cumulLoss/100)
    cumulLoss = 0
  end
end

print('All done!')
