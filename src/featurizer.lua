local featurizer = {}

local function featurizer.makeInput(field)
  local input = torch.Tensor(field:size(1), field:size(2), 11):zero()
  local fieldView = field:view(field:size(1), field:size(2), 1)
  input:scatter(3, fieldView, 1)
  return input
end

return featurizer
