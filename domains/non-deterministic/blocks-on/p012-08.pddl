(define (problem blocks-012-08)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b7) (clear b9) (handempty) (on b0 b3) (on b10 b4) (on b2 b0) (on b3 b11) (on b4 b8) (on b6 b10) (on b7 b5) (on b8 b2) (on b9 b6) (ontable b1) (ontable b11) (ontable b5))
    (:goal (on b11 b6))
)