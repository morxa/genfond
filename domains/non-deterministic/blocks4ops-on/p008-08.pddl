(define (problem blocks-008-08)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7)
    (:init (clear b2) (clear b6) (clear b7) (handempty) (on b1 b5) (on b2 b3) (on b3 b1) (on b5 b4) (on b7 b0) (ontable b0) (ontable b4) (ontable b6))
    (:goal (on b4 b2))
)