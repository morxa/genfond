(define (problem blocks-012-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b7) (handempty) (on b0 b11) (on b1 b5) (on b10 b2) (on b11 b4) (on b2 b9) (on b3 b10) (on b4 b6) (on b6 b8) (on b8 b3) (on b9 b1) (ontable b5) (ontable b7))
    (:goal (and (on b4 b11) (on b11 b8)))
)