(define (problem blocks-010-06)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b4) (clear b6) (clear b7) (handempty) (on b0 b5) (on b1 b2) (on b2 b0) (on b4 b9) (on b5 b8) (on b6 b3) (on b9 b1) (ontable b3) (ontable b7) (ontable b8))
    (:goal (on b8 b7))
)