require 'Player'

local player = Player(10, 10)

print(player.field)
print(player.visible)

player:trigger(torch.Tensor{1,1})
print(player.visible)

player:flag(torch.Tensor{6,6})
print(player.visible)
