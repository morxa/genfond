(define (problem problem-7-5)
    (:domain barman)
    (:objects cocktail1 cocktail2 cocktail3 cocktail4 cocktail5 cocktail6 cocktail7 - cocktail dispenser1 dispenser2 dispenser3 - dispenser left right - hand ingredient1 ingredient2 ingredient3 - ingredient l0 l1 l2 - level shaker1 - shaker shot1 shot2 shot3 shot4 shot5 shot6 shot7 - shot)
    (:init (clean shaker1) (clean shot1) (clean shot2) (clean shot3) (clean shot4) (clean shot5) (clean shot6) (clean shot7) (cocktail-part1 cocktail1 ingredient2) (cocktail-part1 cocktail2 ingredient3) (cocktail-part1 cocktail3 ingredient1) (cocktail-part1 cocktail4 ingredient1) (cocktail-part1 cocktail5 ingredient3) (cocktail-part1 cocktail6 ingredient3) (cocktail-part1 cocktail7 ingredient3) (cocktail-part2 cocktail1 ingredient1) (cocktail-part2 cocktail2 ingredient1) (cocktail-part2 cocktail3 ingredient2) (cocktail-part2 cocktail4 ingredient2) (cocktail-part2 cocktail5 ingredient1) (cocktail-part2 cocktail6 ingredient1) (cocktail-part2 cocktail7 ingredient2) (dispenses dispenser1 ingredient1) (dispenses dispenser2 ingredient2) (dispenses dispenser3 ingredient3) (empty shaker1) (empty shot1) (empty shot2) (empty shot3) (empty shot4) (empty shot5) (empty shot6) (empty shot7) (handempty left) (handempty right) (next l0 l1) (next l1 l2) (ontable shaker1) (ontable shot1) (ontable shot2) (ontable shot3) (ontable shot4) (ontable shot5) (ontable shot6) (ontable shot7) (shaker-empty-level shaker1 l0) (shaker-level shaker1 l0))
    (:goal (and (contains shot1 cocktail4) (contains shot2 cocktail3) (contains shot3 cocktail6) (contains shot4 cocktail1) (contains shot5 cocktail2) (contains shot6 cocktail7) (contains shot7 cocktail5)))
)