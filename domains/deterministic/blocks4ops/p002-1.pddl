(define (problem blocks-002-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1)
    (:init (clear b1) (handempty) (on b1 b0) (ontable b0))
    (:goal (and (on b1 b0)))
)