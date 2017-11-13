require 'nn'
require 'player'

local optim = require 'optim'
local models = require 'models'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training Lifesaver - a neural Minesweeper')
cmd:text()
cmd:text('Options')
cmd:option('-size', 15, 'The size of the field')
cmd:option('-bombs', 40, 'Number of bombs')
cmd:option('-games', 1000, 'Number of games to play')
cmd:option('-type', 0, '0: random, 1: fix, 2: easy corners')
cmd:text()

local opts = cmd:parse(arg)

if opts.type == 2 then
  print('Warning: Hijacking size and number of bombs for easy corners!')
  opts.size = 3
  opts.bombs = 1
end

cmd:log('log.txt', opts)

local size = opts.size
local numBombs = opts.bombs
local threshold = opts.t
local numCells = size*size

local player = Player(size, numBombs, opts.type)

-- TODO make cnn
local net = models.ffnn(opts.loss, numCells, numCells, 2*numCells)

local params, gradParams = net:getParameters()
local outfh = io.open('outputs.txt', 'w')

-- Shortcuts
--local field   = player.field
--local bombs   = player.bombs
--local visible = player.visible
--local flags   = player.flags

print('## Architecture:')
print(net)

local function train()
  local done = false
  local sumLoss = 0
  local rounds = 0
  while not done do
    local input = {visible:view(-1), flags:view(-1)}
    if opts.norm then
      input[1] = input[1]:clone():add(1):div(9)
    end
    local invis = torch.eq(visible, player.INVIS)
    local mask = torch.cbitxor(invis,flags:byte())

    if mask:sum() == 0 then -- everything is revealed or flagged
      done = true
    else
      rounds = rounds + 1
      -- Forward backward
      local output = net:forward(input)
      local target = bombs:view(-1)
      if opts.loss == 'Margin' then
        target = target:clone()
        target[torch.eq(bombs,0)] = -1
      end
      sumLoss = sumLoss + criterion:forward(output, target)
      local gradOutput = criterion:backward(output, target)
      net:backward(input, gradOutput)

      -- Log
      outfh:write('Round ' .. rounds .. '\n')
      outfh:write(field2string(visible) .. '\n\n')
      outfh:write(field2string(output:view(size,size)) .. '\n\n')
      local weights = net:get(#net-1).weight
      outfh:write(tostring(weights))
      
      -- Play the game (with a little help from oracle)
      local unflaggedBombs = torch.cbitxor(bombs:byte(),flags:byte())
      -- TODO Finish fixing this for Margin loss
      if opts.loss = 'Margin' then
        unflaggedBombs[torch.eq(unflaggedBombs,0)] = -10
      end
      local ufbScores = torch.cmul(unflaggedBombs:view(-1):double(),output)
      local max,argmax = ufbScores:max(1)
      local mask = torch.cbitxor(invis,bombs:byte())
      if max[1] >= threshold or mask:sum() == 0 then
        player:flag(argmax[1])
        print('flagging ' .. argmax[1])
      else
        player:triggerMin(output)
        print('trigger min')
      end
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
      done = true
    end
  end
end

local wins = 0
local currWins = 0
local cumulLoss = 0
local config = {learningRate=0.01} -- optim config
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
