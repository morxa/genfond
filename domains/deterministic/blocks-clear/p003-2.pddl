(define (problem blocks-003-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2)
    (:init (clear b1) (handempty) (on b0 b2) (on b1 b0) (ontable b2))
    (:goal (clear b0))
)