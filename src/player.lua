--[[ Player module to start minesweeper games ]]

require('point')

local Player = torch.class('Player')

Player.INVIS = -1
Player.BOMB = -2
Player.FLAG = -3 -- TODO temporary representation

function Player:__init(size, numBombs, type)
  assert(size, 'please specify a size')
  assert(numBombs < size^2, 'too many bombs')
  
  self.size = size
  self.numBombs = numBombs or size
  self.type = type or 0

  self.field = torch.Tensor(size,size)
  self.visible = torch.Tensor(size,size)
  self.bombs = torch.Tensor(size,size)
  self.flags = torch.Tensor(size,size) 
  self.numFlags = 0
end

function Player:printField(field)
  field = field or self.field
  local s = tostring(field)
    :gfind('(.+)\n%[')()
    :gsub('^','  ')
    :gsub('\n','\n  ')
    :gsub('(%d+%.%d)%d*', '%1') -- keep 1 digit only
  print(s)
end

function Player:placeBombs(fix)
  local size = self.size
  local numBombs = self.numBombs
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
      self:placeBombs(false)
    elseif self.type == 1 then
      self:placeBombs(true)
    elseif self.type == 2 then
      self:easyCorners()
    else
      error('unknown type')
    end

    -- place numbers around
    local point = Point(1,1)
    for i=1,size do
      point.y = i
      for j=1,size do
        point.x = j
        self:placeNumber(point)
      end
    end
  end
end

function Player:placeNumber(point)
  if self.field[point.x][point.y] == self.BOMB then 
    return 0
  end

  local number = self:lookAround(point, function(p)
    return (self.field[p.x][p.y] == self.BOMB) and 1 or 0
  end)
  self.field[point.x][point.y] = number
  return number
end

function Player:checkLoc(point)
  return point.x > 0 and point.x <= self.size 
     and point.y > 0 and point.y <= self.size
end

function Player:lookAround(point, func)
  local function apply(p)
    if self:checkLoc(p) then
      return func(p)
    else
      return 0
    end
  end

  local res = 0
  local p = Point(point.x, point.y) -- create a new pointer
  p:left();  res = res + apply(p)
  p:down();  res = res + apply(p)
  p:right(); res = res + apply(p)
  p:right(); res = res + apply(p)
  p:up();    res = res + apply(p)
  p:up();    res = res + apply(p)
  p:left();  res = res + apply(p)
  p:left();  res = res + apply(p)

  return res
end

function Player:reveal(point)
  local isZero  = self.field[point.x][point.y] == 0
  local isInvis = self.visible[point.x][point.y] == self.INVIS
  local isFlag  = self.flags[point.x][point.y] == 1
  if isZero and isInvis and not isFlag then
    self.visible[point.x][point.y] = self.field[point.x][point.y]
    self:lookAround(point, function(p)
      self:reveal(p)
      return 0
    end)
  elseif isInvis and not isFlag then
    self.visible[point.x][point.y] = self.field[point.x][point.y]
  end
end

function Player:trigger(point)
  if self.visible[point.x][point.y] ~= self.INVIS then
    error('ERROR: Revealing an already revealed cell (' .. point.x .. ', ' .. point.y .. ')')
  end
  --print('Revealing ' .. x .. ' ' .. y)
  self:reveal(point)
  return self.field[point.x][point.y] == self.BOMB
end

function Player:flag(point)
  if self.visible[point.x][point.y] ~= self.INVIS then
    error('ERROR: Flagging a visible cell (' .. point.x .. ', ' .. point.y .. ')')
  end
  self.flags[point.x][point.y] = 1
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

