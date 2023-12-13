(define (problem tireworld-085-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l9 - location)
    (:init (road l0 l3) (road l0 l64) (road l1 l44) (road l1 l47) (road l10 l55) (road l10 l79) (road l11 l35) (road l12 l44) (road l12 l49) (road l12 l5) (road l13 l33) (road l13 l63) (road l14 l15) (road l14 l56) (road l15 l14) (road l15 l2) (road l16 l19) (road l16 l3) (road l16 l78) (road l17 l62) (road l17 l74) (road l18 l52) (road l18 l82) (road l19 l16) (road l19 l7) (road l19 l76) (road l2 l15) (road l2 l50) (road l20 l36) (road l20 l43) (road l21 l59) (road l21 l77) (road l22 l60) (road l22 l61) (road l23 l38) (road l23 l57) (road l24 l36) (road l24 l78) (road l25 l34) (road l25 l70) (road l25 l8) (road l26 l51) (road l26 l69) (road l27 l60) (road l27 l67) (road l28 l65) (road l28 l82) (road l29 l69) (road l29 l8) (road l3 l0) (road l3 l16) (road l3 l41) (road l30 l53) (road l30 l54) (road l31 l83) (road l32 l34) (road l32 l43) (road l33 l13) (road l33 l46) (road l34 l25) (road l34 l32) (road l35 l11) (road l35 l65) (road l36 l20) (road l36 l24) (road l37 l5) (road l37 l68) (road l38 l23) (road l38 l50) (road l39 l52) (road l39 l9) (road l4 l55) (road l4 l58) (road l40 l79) (road l40 l81) (road l41 l3) (road l41 l59) (road l42 l75) (road l42 l8) (road l43 l20) (road l43 l32) (road l44 l1) (road l44 l12) (road l45 l73) (road l45 l9) (road l46 l33) (road l46 l80) (road l47 l1) (road l47 l75) (road l48 l56) (road l48 l70) (road l49 l12) (road l49 l58) (road l5 l12) (road l5 l37) (road l50 l2) (road l50 l38) (road l51 l26) (road l51 l77) (road l52 l18) (road l52 l39) (road l53 l30) (road l53 l84) (road l54 l30) (road l54 l57) (road l55 l10) (road l55 l4) (road l56 l14) (road l56 l48) (road l56 l62) (road l57 l23) (road l57 l54) (road l58 l4) (road l58 l49) (road l59 l21) (road l59 l41) (road l6 l72) (road l6 l84) (road l60 l22) (road l60 l27) (road l61 l22) (road l61 l74) (road l62 l17) (road l62 l56) (road l63 l13) (road l63 l73) (road l64 l0) (road l64 l66) (road l65 l28) (road l65 l35) (road l66 l64) (road l66 l80) (road l67 l27) (road l67 l7) (road l68 l37) (road l68 l83) (road l69 l26) (road l69 l29) (road l7 l19) (road l7 l67) (road l70 l25) (road l70 l48) (road l71 l72) (road l71 l81) (road l72 l6) (road l72 l71) (road l73 l45) (road l73 l63) (road l74 l17) (road l74 l61) (road l75 l42) (road l75 l47) (road l76 l19) (road l76 l8) (road l77 l21) (road l77 l51) (road l78 l16) (road l78 l24) (road l79 l10) (road l79 l40) (road l8 l25) (road l8 l29) (road l8 l42) (road l8 l76) (road l80 l46) (road l80 l66) (road l81 l40) (road l81 l71) (road l82 l18) (road l82 l28) (road l83 l31) (road l83 l68) (road l84 l53) (road l84 l6) (road l9 l39) (road l9 l45) (spare-in l0) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l18) (spare-in l2) (spare-in l21) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l33) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l77) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l9) (vehicle-at l11))
    (:goal (vehicle-at l31))
)