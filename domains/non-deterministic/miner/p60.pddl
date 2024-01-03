(define (problem miner-60)
(:domain miner)
(:objects 
	L11 - location
	L12 - location
	L13 - location
	L21 - location
	L22 - location
	L23 - location
	L31 - location
	L32 - location
	L33 - location

	r1 - rock
)
(:init
	(person-alive)
	(person-at L11)
	(botton-loc L11)

	(rock-at r1 L12)

	(gold-bad-at L11)
	(gold-bad-at L12)

	(gold-good-at L32)
	(gold-good-at L31)

	(road L11 L21)
	(road L11 L12)
	(road L12 L22)
	(road L12 L13)
	(road L12 L11)
	(road L21 L11)
	(road L21 L31)
	(road L21 L22)
	(road L22 L12)
	(road L22 L32)
	(road L22 L23)
	(road L22 L21)
	(road L31 L21)
	(road L31 L32)
	(road L32 L22)
	(road L32 L33)
	(road L32 L31)
)
(:goal (and (person-alive) (goldcount g2)))
)