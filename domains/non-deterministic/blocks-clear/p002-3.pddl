(define (problem blocks-002-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1)
    (:init (clear b0) (handempty) (on b0 b1) (ontable b1))
    (:goal (clear b1))
)