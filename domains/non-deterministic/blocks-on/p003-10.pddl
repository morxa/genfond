(define (problem blocks-003-10)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2)
    (:init (clear b0) (clear b2) (handempty) (on b0 b1) (ontable b1) (ontable b2))
    (:goal (on b2 b0))
)