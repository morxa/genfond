(define (problem blocks-004-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3)
    (:init (clear b1) (clear b2) (handempty) (on b1 b3) (on b3 b0) (ontable b0) (ontable b2))
    (:goal (clear b0))
)