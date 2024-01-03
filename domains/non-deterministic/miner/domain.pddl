; Domain proposed by Tomas Geffner and Hector Geffner

(define (domain miner)
  (:requirements :typing :strips :non-deterministic)
  (:types location rock goldcount)
  (:constants g1 g2 g3 - goldcount)
  (:predicates (person-at ?loc - location)
               (botton-loc ?loc - location)
               (person-alive)
               (rock-at ?r - rock ?loc - location)
               (road ?from - location ?to - location)
               (holding ?r - rock)
               (gold-bad-at ?loc - location)
               (gold-good-at ?loc - location)
               (goldcount ?gc - goldcount)
               (botton-pressed))

  (:action move-person
    :parameters (?from - location ?to - location)
    :precondition (and (person-at ?from) (road ?from ?to) (person-alive))
    :effect (and (person-at ?to) (not (person-at ?from)))
  )

  (:action pick-rock
    :parameters (?loc - location ?r - rock)
    :precondition (and (person-at ?loc) (rock-at ?r ?loc) (person-alive))
    :effect (and (not (rock-at ?r ?loc)) (holding ?r))
  )

  (:action drop-rock
    :parameters (?loc - location ?r - rock)
    :precondition (and (person-at ?loc) (holding ?r) (person-alive))
    :effect (and (rock-at ?r ?loc) (not (holding ?r)))
  )

  (:action drop-rock-press
    :parameters (?loc - location ?r - rock)
    :precondition (and (person-at ?loc) (holding ?r) (person-alive) (botton-loc ?loc))
    :effect (and (rock-at ?r ?loc) (botton-pressed) (not (holding ?r)))
  )

  (:action pick-bad-gold-1
    :parameters (?loc - location)
    :precondition (and (person-at ?loc) (gold-bad-at ?loc) (person-alive) (not (goldcount g1)))
    :effect (oneof (not (person-alive))
            (and (goldcount g1) (not (gold-bad-at ?loc))))
  )

  (:action pick-bad-gold-2
    :parameters (?loc - location)
    :precondition (and (person-at ?loc) (gold-bad-at ?loc) (person-alive) (goldcount g1) (not  (goldcount g2)))
    :effect (oneof (not (person-alive))
            (and (goldcount g2) (not (gold-bad-at ?loc))))
  )

  (:action pick-bad-gold-3
    :parameters (?loc - location)
    :precondition (and (person-at ?loc) (gold-bad-at ?loc) (person-alive) (goldcount g2) (not (goldcount g3)))
    :effect (oneof (not (person-alive))
            (and (goldcount g3) (not (gold-bad-at ?loc))))
  )

  (:action pick-good-gold-1
    :parameters (?loc - location)
    :precondition (and (person-at ?loc) (gold-good-at ?loc) (person-alive) (not (goldcount g1)) (botton-pressed))
    :effect (and (goldcount g1) (not (gold-good-at ?loc)))
  )

  (:action pick-good-gold-2
    :parameters (?loc - location)
    :precondition (and (person-at ?loc) (gold-good-at ?loc) (person-alive) (goldcount g1) (not (goldcount g2)) (botton-pressed))
    :effect (and (goldcount g2) (not (gold-good-at ?loc)))
  )

  (:action pick-good-gold-3
    :parameters (?loc - location)
    :precondition (and (person-at ?loc) (gold-good-at ?loc) (person-alive) (goldcount g2) (not (goldcount g3)) (botton-pressed))
    :effect (and (goldcount g3) (not (gold-good-at ?loc)))
  )

)
