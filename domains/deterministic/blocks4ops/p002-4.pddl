(define (problem blocks-002-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1)
    (:init (clear b1) (handempty) (on b1 b0) (ontable b0))
    (:goal (and (clear b1) (on b1 b0) (ontable b0)))
)