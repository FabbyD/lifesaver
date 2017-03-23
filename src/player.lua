--[[ Player module to start minesweeper games ]]

local Player = torch.class('Player')

Player.INVIS = -1
Player.BOMB = -2
Player.FLAG = -2 -- TODO temporary representation

function Player:__init(size, numBombs, type)
  assert(size, 'please specify a size')
  assert(numBombs <= size^2, 'too many bombs')
  
  self.size = size
  self.numBombs = numBombs or size
  self.type = type

  self.field = torch.Tensor(size,size)
  self.visible = torch.Tensor(size,size)
  self.bombs = torch.Tensor(size,size)
  self.flags = torch.Tensor(size,size) 
  self.numFlags = 0
end

function Player:randomField(fix)
  if fix then
    torch.manualSeed(0)
  end
  local shuffle = torch.randperm(size*size)
  for i=1,size*size do
    local j = math.floor((i-1)/size)+1
    local k = (i-1)%size+1
    local isbomb = shuffle[i]>0 and shuffle[i]<=numBombs
    if isbomb then
      self.field[j][k] = self.BOMB
      self.bombs[j][k] = 1
    end
  end
end

function Player:easyCorners()
  local corner = math.random(0,3)
  local i = math.floor(corner/2)*(self.size-1)+1
  local j = (corner%2)*(self.size-1)+1
  self.field[i][j] = self.BOMB
  self.bombs[i][j] = 1
end

function Player:resetField()
  local size = self.size
  local numBombs = self.numBombs
  self.field:zero()
  self.bombs:zero()
  self.flags:zero()
  self.visible:fill(self.INVIS)
  self.numFlags = 0
  if numBombs > 0 then
    -- place bombs
    if self.type == 0 then
      self:randomField(false)
    elseif self.type == 1 then
      self:randomField(true)
    elseif self.type == 2 then
      self:easyCorners()
    else
      error('unknown type')
    end

    -- place numbers around
    local loc = torch.Tensor(2)
    for i=1,size do
      loc[1] = i
      for j=1,size do
        loc[2] = j
        self:placeNumber(loc)
      end
    end
  end
end

function Player:placeNumber(loc)
  local x,y = loc[1], loc[2]
  if self.field[x][y] == self.BOMB then 
    return 0
  end

  local number = self:lookAround(loc, function(l)
    local x,y = l[1],l[2]
    return (self.field[x][y] == self.BOMB) and 1 or 0
  end)
  self.field[x][y] = number
  return number
end

function Player:checkLoc(loc)
  return loc[1] > 0 and loc[1] <= self.size 
     and loc[2] > 0 and loc[2] <= self.size
end

function Player:lookAround(loc, func)
  local function apply(l)
    if self:checkLoc(l) then
      return func(l)
    else
      return 0
    end
  end

  local res = 0
  local nloc = loc:clone() -- create a new pointer
  self:left(nloc);  res = res + apply(nloc)
  self:down(nloc);  res = res + apply(nloc)
  self:right(nloc); res = res + apply(nloc)
  self:right(nloc); res = res + apply(nloc)
  self:up(nloc);    res = res + apply(nloc)
  self:up(nloc);    res = res + apply(nloc)
  self:left(nloc);  res = res + apply(nloc)
  self:left(nloc);  res = res + apply(nloc)

  return res
end

function Player:up(loc)
  loc[1] = loc[1] - 1
  return loc
end

function Player:down(loc)
  loc[1] = loc[1] + 1
  return loc
end

function Player:left(loc)
  loc[2] = loc[2] - 1
  return loc
end

function Player:right(loc)
  loc[2] = loc[2] + 1
  return loc
end

function Player:reveal(loc)
  local x,y = loc[1],loc[2]
  local isZero  = self.field[x][y] == 0
  local isInvis = self.visible[x][y] == self.INVIS
  local isFlag  = self.flags[x][y] == 1
  if isZero and isInvis and not isFlag then
    self.visible[x][y] = self.field[x][y]
    self:lookAround(loc, function(l)
      self:reveal(l)
      return 0
    end)
  elseif isInvis and not isFlag then
    self.visible[x][y] = self.field[x][y]
  end
end

function Player:loc2xy(loc)
  local x,y
  if torch.isTensor(loc) then
    x,y = loc[1], loc[2]
  else
    x = math.ceil(loc/self.size)
    y = (loc-1)%self.size+1
    loc = torch.Tensor{x,y}
  end
  return loc, x, y
end

function Player:trigger(loc)
  local loc,x,y = self:loc2xy(loc)
  if self.visible[x][y] ~= self.INVIS then
    error('ERROR: Revealing an already revealed cell (' .. x .. ', ' .. y .. ')')
  end
  --print('Revealing ' .. x .. ' ' .. y)
  self:reveal(loc)
  return self.field[x][y] == self.BOMB
end

function Player:flag(loc)
  local loc,x,y = self:loc2xy(loc)
  if self.visible[x][y] ~= self.INVIS then
    error('ERROR: Flagging a visible cell (' .. x .. ', ' .. y .. ')')
  end
  self.flags[x][y] = 1
  self.numFlags = self.numFlags + 1
  --print('Flagging ' .. x .. ' ' .. y)
end

function Player:triggerMin(scores)
  -- Ignore bombs and revealed cells in argmin
  local invis = torch.ne(self.visible,self.INVIS)
  local mask = torch.cbitor(self.bombs:byte(), invis)
  --print('visible:')
  --print(self.visible)
  --print(invis)
  --print('bombs:')
  --print(self.bombs:byte())
  --print(mask)
  scores[mask] = 1
  local min, argmin = scores:min(1)
  self:trigger(argmin[1])
end

