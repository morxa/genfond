(define (problem problem-7-1)
    (:domain barman)
    (:objects cocktail1 cocktail2 cocktail3 cocktail4 cocktail5 cocktail6 cocktail7 - cocktail dispenser1 dispenser2 dispenser3 - dispenser left right - hand ingredient1 ingredient2 ingredient3 - ingredient l0 l1 l2 - level shaker1 - shaker shot1 shot2 - shot)
    (:init (clean shaker1) (clean shot1) (clean shot2) (cocktail-part1 cocktail1 ingredient2) (cocktail-part1 cocktail2 ingredient1) (cocktail-part1 cocktail3 ingredient3) (cocktail-part1 cocktail4 ingredient3) (cocktail-part1 cocktail5 ingredient3) (cocktail-part1 cocktail6 ingredient1) (cocktail-part1 cocktail7 ingredient1) (cocktail-part2 cocktail1 ingredient1) (cocktail-part2 cocktail2 ingredient2) (cocktail-part2 cocktail3 ingredient1) (cocktail-part2 cocktail4 ingredient1) (cocktail-part2 cocktail5 ingredient2) (cocktail-part2 cocktail6 ingredient3) (cocktail-part2 cocktail7 ingredient3) (dispenses dispenser1 ingredient1) (dispenses dispenser2 ingredient2) (dispenses dispenser3 ingredient3) (empty shaker1) (empty shot1) (empty shot2) (handempty left) (handempty right) (next l0 l1) (next l1 l2) (ontable shaker1) (ontable shot1) (ontable shot2) (shaker-empty-level shaker1 l0) (shaker-level shaker1 l0))
    (:goal (and (contains shot1 cocktail3) (contains shot2 cocktail2) (contains shot3 cocktail6) (contains shot4 cocktail5) (contains shot5 cocktail1) (contains shot6 cocktail7) (contains shot7 cocktail4)))
)