--[[ Player module to start minesweeper games ]]

local Player = torch.class('Player')

Player.bomb = -1

function Player:__init(size, numBombs)
  assert(size, 'please specify a size')
  assert(numBombs <= size^2, 'too many bombs')

  self.size = size
  self.numBombs = numBombs or size
  self.field = torch.zeros(size,size)

  -- Initialize the field
  self:initField()
end

function Player:initField()
  local size = self.size
  local numBombs = self.numBombs
  self.field:zero()
  if numBombs > 0 then
    local shuffle = torch.randperm(size*size)
    for i=1,size*size do
      local j = math.floor((i-1)/size)+1
      local k = (i-1)%size+1
      local isbomb = shuffle[i]>0 and shuffle[i]<=numBombs
      self.field[j][k] = isbomb and self.bomb or 0
    end

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
  if self.field[x][y] == self.bomb then 
    return 0
  end

  local number = self:lookAround(loc, function(l)
    local x,y = l[1],l[2]
    return (self.field[x][y] == self.bomb) and 1 or 0
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