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

local function printField(field)
  local str = tostring(field):gfind('(.+)\n%[')():gsub('^','  '):gsub('\n','\n  ')
  print(str)
end

local params, gradParams = net:getParameters()

local function play()
  local done = false
  local lost = false
  local sumLoss = 0
  while not done do
    --printField(visible)
    local flagsView = flags:view(-1)
    local input = visible:view(-1) -- flatten
    local iMask = torch.eq(input, player.INVIS)
    local fMask = torch.eq(flagsView, 1)
    local mask = torch.cbitxor(iMask,fMask)
    if mask:sum() == 0 then -- everything is revealed or flagged
      done = true
    else
      local output = net:forward(input)
      local target = bombs:view(-1)
      local loss = criterion:forward(output, target)
      sumLoss = sumLoss + loss
      local gradOutput = criterion:backward(output, target)
      net:backward(input, gradOutput)
      local invisOutput = output[mask]
      local max, argmax = invisOutput:max(1)
      local min, argmin = invisOutput:min(1)
      --print(string.format('Max: %.2f;  Min: %.2f;  Loss: %.2f', max[1], min[1], loss))
      if max[1] >= threshold then
        local bombloc = findNthElem(mask, argmax[1], 1)
        player:flag(bombloc)
      else
        local safeloc = findNthElem(mask, argmin[1], 1)
        lost = player:trigger(safeloc)
        done = lost
      end
    end
  end
  --print('Done with:')
  --printField(visible)
  --print('')
  return lost, sumLoss
end

local numGames = 1000
local wins = 0
local currWins = 0
for g=1,numGames do
  player:resetField()
  --print('### Game ' .. g)
  --print('===== Field ======')
  --printField(field)
  --print('==================')

  local function feval(x)
    if x ~= params then
      print('WARNING: Copying parameters')
      params:copy(x)
    end

    gradParams:zero()

    local lost, loss = play()
    if not lost then
      wins = wins + 1
      currWins = currWins + 1
    end

    return loss, gradParams
  end
  optim.sgd(feval, params, config)

  if g%100==0 then
    print('game ' .. g .. ' wins: ' .. currWins)
    currWins = 0
  end
end

print('All done!')
print(string.format('Win ratio: %d/%d = %.2f', wins, numGames, 100*wins/numGames))
