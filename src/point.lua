--[[ Tile module inside mine field ]]

local Point = torch.class('Point')

function Point:__init(x, y)
  x = x or 1
  y = y or 1
  self.x = x
  self.y = y
end

function Point:up()
  self.x = self.x - 1
end

function Point:down()
  self.x = self.x + 1
end

function Point:left()
  self.y = self.y - 1
end

function Point:right()
  self.y = self.y + 1
end
